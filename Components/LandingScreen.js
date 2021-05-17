import React from 'react';
import { View, Text, Button,StyleSheet, TouchableOpacity } from 'react-native';
 
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#003f5c',
    alignItems: 'center',
    justifyContent: 'center',
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
  logo:{
    fontWeight:"bold",
    fontSize:50,
    color:"#fb5b5a",
    marginBottom:40
  },
  White:{
    fontSize: 19,
    color : '#ffffff'
  },
});
 
const LandingScreen = ({navigation}) => {
  return (
    <View style={styles.container}>
      <Text style ={styles.logo}>CRE </Text>
      <View style = {styles.loginBtn}>
      <TouchableOpacity onPress = {() => navigation.navigate('SignIn')}>
        <Text style = {styles.White}>Go to SignIn</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};
 
export default LandingScreen;