import React from 'react';
import { View, Text, Button, StyleSheet, TextInput, Keyboard, TouchableOpacity } from 'react-native';
import { color } from 'react-native-reanimated';
import { Colors } from 'react-native/Libraries/NewAppScreen';

 
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#003f5c',
    alignItems: 'center',
    justifyContent: 'center',
  },
  inputView: {
    width:"80%",
    backgroundColor:"#465881",
    borderRadius:25,
    height:60,
    marginBottom:10,
    justifyContent:"center",
    padding:20
  },
  inputText:{
    height:55,
    color:"white"
  },
  loginBtn:{
    width:"80%",
    backgroundColor:"#fb5b5a",
    borderRadius:25,
    height:50,
    alignItems:"center",
    justifyContent:"center",
    marginTop:10,
    marginBottom:10
  },
  White:{
    fontSize: 18,
    color : '#ffffff'
  },
  error: {
    marginBottom: 10,
    marginTop: 2,
    color: "white",
    fontSize: 13,
  },
});
 
const SignUpScreen = ({ onSignUp, err, navigation }) => {
  const[firstName, setFirstName] = React.useState('');
  const[lastName, setLastName] = React.useState('');
  const[zipcode, setZipCode] = React.useState('');
  const[email, setEmail] = React.useState('');
  const[password, setPassword] = React.useState('');
  
  return (
    <View style={styles.container}>
      <Text style={styles.error}>{err && `Registration failed`}</Text>
      <View style = {styles.inputView}>
        <TextInput
         placeholder="First Name"
         placeholderTextColor="#003f5c"
         style={styles.inputText}
         onBlur={Keyboard.dismiss}
         value= {firstName}
         onChange={event => setFirstName(event.nativeEvent.text)}
        />
      </View>
      <View style = {styles.inputView}>
        <TextInput
         placeholder="Last Name"
         placeholderTextColor="#003f5c"
         style={styles.inputText}
         onBlur={Keyboard.dismiss}
         value= {lastName}
         onChange={event => setLastName(event.nativeEvent.text)} 
        />
      </View>
      <View style = {styles.inputView}>
        <TextInput
         placeholder="Zipcode"
         placeholderTextColor="#003f5c"
         style={styles.inputText}
         onBlur={Keyboard.dismiss}
         value= {zipcode}
         onChange={event => setZipCode(event.nativeEvent.text)} 
        />
      </View>
      <View style = {styles.inputView}>
        <TextInput
         placeholder="Email"
         placeholderTextColor="#003f5c"
         style={styles.inputText}
         onBlur={Keyboard.dismiss}
         value= {email}
         autoCapitalize='none'
         autoCorrect={false}
         onChange={event => setEmail(event.nativeEvent.text)}
        />
      </View>
      <View style = {styles.inputView}>
        <TextInput
         placeholder="Password"
         placeholderTextColor="#003f5c"
         style={styles.inputText}
         onBlur={Keyboard.dismiss}
         secureTextEntry={true}
         value= {password}
         onChange={event => setPassword(event.nativeEvent.text)} 
        />
      </View>
      <View style = {styles.loginBtn}>
        <TouchableOpacity  onPress = {() => onSignUp(firstName, lastName,zipcode, email, password)}>
        <Text style = {styles.White} >SIGN UP</Text>
        </TouchableOpacity>
      </View>
      <View>
        <TouchableOpacity  onPress = {() => navigation.navigate('SignIn')}>
        <Text style = {styles.White} >ALREADY HAVE AN ACCOUNT?</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};
 
export default SignUpScreen;