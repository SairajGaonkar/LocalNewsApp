import React from 'react';
import { View, Text, Button, StyleSheet, TextInput, Keyboard, TouchableOpacity, } from 'react-native';
import { color, event } from 'react-native-reanimated';
 
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
    marginBottom:20,
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
    marginTop:0,
    marginBottom:10
  },
  signUpBtn:{
    width:"80%",
    backgroundColor:"#fb5b5a",
    borderRadius:25,
    height:50,
    alignItems:"center",
    justifyContent:"center",
    marginTop:15,
    marginBottom:10
  },
  White:{
    fontSize: 19,
    color : '#ffffff'
  },
  error: {
    marginBottom: 10,
    marginTop: 2,
    color: "white",
    fontSize: 13,
  },
});




const SignInScreen = ({onSignIn,err, navigation}) => {
  const[email, setEmail] = React.useState('');
  const[password, setPassword] = React.useState('');
  
  return (
    <View style={styles.container}>
      <Text style={styles.error}>{err && `Invalid credentials`}</Text>
      <View style={styles.inputView}>
      <TextInput
        placeholder="Email"
        placeholderTextColor="#003f5c"
        style={styles.inputText}
        onBlur={Keyboard.dismiss}
        value= {email}
        autoCapitalize = 'none'
        autoCorrect={false}
        onChange={event => setEmail(event.nativeEvent.text)}
      />
      </View>
      <View style={styles.inputView}>
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
        <TouchableOpacity onPress = {() => onSignIn(email,password)}>
        <Text style = {styles.White} >LOGIN</Text>
        </TouchableOpacity>
      </View>

      <View>
        <TouchableOpacity >
        <Text style = {styles.White} >Forgot Password? </Text>
        </TouchableOpacity>
      </View>

      <View style = {styles.signUpBtn}>
        <TouchableOpacity onPress = {() => navigation.navigate('SignUp')}>
          <Text style = {styles.White}>CREATE ACCOUNT</Text>
        </TouchableOpacity>
      </View>

    </View>
  );
};
 
export default SignInScreen;